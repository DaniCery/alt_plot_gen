from alt_plot_gen.data_sources.local_disk import import_dataset
import regex as re
import os


def clean_plot(plot):
    """
    clean Plot column by removing non alphanumeric and special characters
    """
    plot = plot.lower()
    plot = re.sub(r"what's", "what is ", plot)
    plot = re.sub(r"\'s", " ", plot)
    plot = re.sub(r"\'ve", " have ", plot)
    plot = re.sub(r"can't", "can not ", plot)
    plot = re.sub(r"n't", " not ", plot)
    plot = re.sub(r"i'm", "i am ", plot)
    plot = re.sub(r"\'re", " are ", plot)
    plot = re.sub(r"\'d", " would ", plot)
    plot = re.sub(r"\'ll", " will ", plot)
    plot = re.sub(r"\'scuse", " excuse ", plot)
    #plot = re.sub('\W', ' ', plot)
    #plot = re.sub('\s+', ' ', plot)
    plot = plot.strip(' ')

    exceptions = [r'\[d+\]']
    rx = re.compile(r'\b(?:{})\b(*SKIP)(*FAIL)|\b[A-Za-z]*\d\w*\b'.format("|".join(exceptions)))
    nstrings = [" ".join(
        filter(None, (rx.sub('', word)
        for word in string.split())))
        for string in [plot]]
    aux = ' '.join(nstrings)
    aux = aux.replace("\\'", "\'")
    aux = re.sub("\'", '"', aux) #replace "\'" for "'"
    aux = re.sub('"', "'", aux)
    plot_cleaned = re.sub("[\(\[].*?[\)\]]", "", aux) #remove contents inside brackets
    plot_cleaned = plot_cleaned.replace(' .','.').replace(' ,',',')\
                               .replace(' ;',';').replace(' ,',',')\
                               .replace(' ?','?').replace(' !','!') #remove white spaces left from prev line of code
    return plot_cleaned

def clean_data():
    """
    clean raw data by removing buggy or irrelevant plots
    or columns for the training set
    """
    # Import df_raw
    df_raw = import_dataset()

    '''
    #Already cleaned and harmonized

    # Clean Plot column
    df_raw['Plot'] = df_raw['Plot'].apply(clean_plot)

    # Harmonize Genre
    df = harmonize_genre(df_raw)
    '''
    df = df_raw

    df = df[df.Plot.notnull()]

    #boolean mask to take plots with more than 50 words: we take all the movies with more than 30 words plots to maximize the train data
    train_set = df[df['Plot'].apply(lambda x: len(x.split()) > 30)]
    #print('train_set length after removing plots less than 30 words length: ',len(train_set))
    #Cut the long plots (after more than 1024 tokens, pre-training gpt2 does not work due to gpt2-medium usage limitations)
    train_set['Plot'] = train_set['Plot'].apply(lambda x: " ".join(x.split()[:350]))  #cut all plots until the 350th word

    #Select only plots with more than 100 words. We will cut 50 ending words later to extract original ending
    test_set = df[df['Plot'].apply(lambda x: len(x.split()) > 100)]
    test_set = test_set[test_set['Plot'].apply(lambda x: len(x.split()) < 400)]

    #Create a very small test set to compare generated text with the reality (plots selectable by the app at the end)
    full_set = df[df['Plot'].apply(lambda x: len(x.split()) > 100)]
    full_set = full_set[full_set['Release Year']>1979] #used by our app at the end (selectable only movies made from 1980)

    test_set = test_set.sample(n = 200)  #used for testing

    #Let's comment out the next line. Since it's not a classification problem, for our text generation project we'll stil maintain the test set values in train_set
    #train_set = train_set.loc[~train_set.index.isin(test_set.index)]

    #For the test set only, keep last 50 words in a new column, then remove them from original column to save the Original Ending
    test_set['True_end_plot'] = test_set['Plot'].str.split().str[-50:].apply(' '.join)
    test_set['Plot'] = test_set['Plot'].str.split().str[:-50].apply(' '.join)

    #to be used by our app at the end
    full_set['True_end_plot'] = full_set['Plot'].str.split().str[-50:].apply(' '.join)
    full_set['Plot'] = full_set['Plot'].str.split().str[:-50].apply(' '.join)

    #Reset the indexes
    full_set = full_set.reset_index()
    test_set = test_set.reset_index()
    df = train_set.reset_index()   #consider training set as main df

    test_path = os.environ.get("LOCAL_DATA_PATH")
    full_set.to_csv(os.path.join(test_path,r'dataset_app.csv'), index = False, header=True)

    return df, test_set


def harmonize_genre(movies):
    # harmonization
    movies['GenreCorrected'] = movies['Genre']
    movies['GenreCorrected']=movies['GenreCorrected'].str.strip()
    movies['GenreCorrected']=movies['GenreCorrected'].str.replace(' - ', '|')
    movies['GenreCorrected']=movies['GenreCorrected'].str.replace(' / ', '|')
    movies['GenreCorrected']=movies['GenreCorrected'].str.replace('/', '|')
    movies['GenreCorrected']=movies['GenreCorrected'].str.replace(' & ', '|')
    movies['GenreCorrected']=movies['GenreCorrected'].str.replace(', ', '|')
    movies['GenreCorrected']=movies['GenreCorrected'].str.replace('; ', '|')
    movies['GenreCorrected']=movies['GenreCorrected'].str.replace('bio-pic', 'biography')
    movies['GenreCorrected']=movies['GenreCorrected'].str.replace('biopic', 'biography')
    movies['GenreCorrected']=movies['GenreCorrected'].str.replace('biographical', 'biography')
    movies['GenreCorrected']=movies['GenreCorrected'].str.replace('biodrama', 'biography')
    movies['GenreCorrected']=movies['GenreCorrected'].str.replace('bio-drama', 'biography')
    movies['GenreCorrected']=movies['GenreCorrected'].str.replace('biographic', 'biography')
    movies['GenreCorrected']=movies['GenreCorrected'].str.replace(' \(film genre\)', '')
    movies['GenreCorrected']=movies['GenreCorrected'].str.replace('animated','animation')
    movies['GenreCorrected']=movies['GenreCorrected'].str.replace('anime','animation')
    movies['GenreCorrected']=movies['GenreCorrected'].str.replace('children\'s','children')
    movies['GenreCorrected']=movies['GenreCorrected'].str.replace('comedey','comedy')
    movies['GenreCorrected']=movies['GenreCorrected'].str.replace('\[not in citation given\]','')
    movies['GenreCorrected']=movies['GenreCorrected'].str.replace(' set 4,000 years ago in the canadian arctic','')
    movies['GenreCorrected']=movies['GenreCorrected'].str.replace('historical','history')
    movies['GenreCorrected']=movies['GenreCorrected'].str.replace('romantic','romance')
    movies['GenreCorrected']=movies['GenreCorrected'].str.replace('3-d','animation')
    movies['GenreCorrected']=movies['GenreCorrected'].str.replace('3d','animation')
    movies['GenreCorrected']=movies['GenreCorrected'].str.replace('viacom 18 motion pictures','')
    movies['GenreCorrected']=movies['GenreCorrected'].str.replace('sci-fi','science_fiction')
    movies['GenreCorrected']=movies['GenreCorrected'].str.replace('ttriller','thriller')
    movies['GenreCorrected']=movies['GenreCorrected'].str.replace('.','')
    movies['GenreCorrected']=movies['GenreCorrected'].str.replace('based on radio serial','')
    movies['GenreCorrected']=movies['GenreCorrected'].str.replace(' on the early years of hitler','')
    movies['GenreCorrected']=movies['GenreCorrected'].str.replace('sci fi','science_fiction')
    movies['GenreCorrected']=movies['GenreCorrected'].str.replace('science fiction','science_fiction')
    movies['GenreCorrected']=movies['GenreCorrected'].str.replace(' (30min)','')
    movies['GenreCorrected']=movies['GenreCorrected'].str.replace('16??mm film','short')
    movies['GenreCorrected']=movies['GenreCorrected'].str.replace('\[140\]','drama')
    movies['GenreCorrected']=movies['GenreCorrected'].str.replace('\[144\]','')
    movies['GenreCorrected']=movies['GenreCorrected'].str.replace(' for ','')
    movies['GenreCorrected']=movies['GenreCorrected'].str.replace('adventures','adventure')
    movies['GenreCorrected']=movies['GenreCorrected'].str.replace('kung fu','martial_arts')
    movies['GenreCorrected']=movies['GenreCorrected'].str.replace('kung-fu','martial_arts')
    movies['GenreCorrected']=movies['GenreCorrected'].str.replace('martial arts','martial_arts')
    movies['GenreCorrected']=movies['GenreCorrected'].str.replace('world war ii','war')
    movies['GenreCorrected']=movies['GenreCorrected'].str.replace('world war i','war')
    movies['GenreCorrected']=movies['GenreCorrected'].str.replace('biography about montreal canadiens star|maurice richard','biography')
    movies['GenreCorrected']=movies['GenreCorrected'].str.replace('bholenath movies|cinekorn entertainment','')
    movies['GenreCorrected']=movies['GenreCorrected'].str.replace(' \(volleyball\)','')
    movies['GenreCorrected']=movies['GenreCorrected'].str.replace('spy film','spy')
    movies['GenreCorrected']=movies['GenreCorrected'].str.replace('anthology film','anthology')
    movies['GenreCorrected']=movies['GenreCorrected'].str.replace('biography fim','biography')
    movies['GenreCorrected']=movies['GenreCorrected'].str.replace('avant-garde','avant_garde')
    movies['GenreCorrected']=movies['GenreCorrected'].str.replace('biker film','biker')
    movies['GenreCorrected']=movies['GenreCorrected'].str.replace('buddy cop','buddy')
    movies['GenreCorrected']=movies['GenreCorrected'].str.replace('buddy film','buddy')
    movies['GenreCorrected']=movies['GenreCorrected'].str.replace('comedy 2-reeler','comedy')
    movies['GenreCorrected']=movies['GenreCorrected'].str.replace('films','')
    movies['GenreCorrected']=movies['GenreCorrected'].str.replace('film','')
    movies['GenreCorrected']=movies['GenreCorrected'].str.replace('biography of pioneering american photographer eadweard muybridge','biography')
    movies['GenreCorrected']=movies['GenreCorrected'].str.replace('british-german co-production','')
    movies['GenreCorrected']=movies['GenreCorrected'].str.replace('bruceploitation','martial_arts')
    movies['GenreCorrected']=movies['GenreCorrected'].str.replace('comedy-drama adaptation of the mordecai richler novel','comedy-drama')
    movies['GenreCorrected']=movies['GenreCorrected'].str.replace('movies by the mob\|knkspl','')
    movies['GenreCorrected']=movies['GenreCorrected'].str.replace('movies','')
    movies['GenreCorrected']=movies['GenreCorrected'].str.replace('movie','')
    movies['GenreCorrected']=movies['GenreCorrected'].str.replace('coming of age','coming_of_age')
    movies['GenreCorrected']=movies['GenreCorrected'].str.replace('coming-of-age','coming_of_age')
    movies['GenreCorrected']=movies['GenreCorrected'].str.replace('drama about child soldiers','drama')
    movies['GenreCorrected']=movies['GenreCorrected'].str.replace('(( based).+)','')
    movies['GenreCorrected']=movies['GenreCorrected'].str.replace('(( co-produced).+)','')
    movies['GenreCorrected']=movies['GenreCorrected'].str.replace('(( adapted).+)','')
    movies['GenreCorrected']=movies['GenreCorrected'].str.replace('(( about).+)','')
    movies['GenreCorrected']=movies['GenreCorrected'].str.replace('musical b','musical')
    movies['GenreCorrected']=movies['GenreCorrected'].str.replace('animationchildren','animation|children')
    movies['GenreCorrected']=movies['GenreCorrected'].str.replace(' period','period')
    movies['GenreCorrected']=movies['GenreCorrected'].str.replace('drama loosely','drama')
    movies['GenreCorrected']=movies['GenreCorrected'].str.replace(' \(aquatics|swimming\)','')
    movies['GenreCorrected']=movies['GenreCorrected'].str.replace(' \(aquatics|swimming\)','')
    movies['GenreCorrected']=movies['GenreCorrected'].str.replace("yogesh dattatraya gosavi's directorial debut \[9\]",'')
    movies['GenreCorrected']=movies['GenreCorrected'].str.replace("war-time","war")
    movies['GenreCorrected']=movies['GenreCorrected'].str.replace("wartime","war")
    movies['GenreCorrected']=movies['GenreCorrected'].str.replace("ww1","war")
    movies['GenreCorrected']=movies['GenreCorrected'].str.replace('unknown','')
    movies['GenreCorrected']=movies['GenreCorrected'].str.replace("wwii","war")
    movies['GenreCorrected']=movies['GenreCorrected'].str.replace('psychological','psycho')
    movies['GenreCorrected']=movies['GenreCorrected'].str.replace('rom-coms','romance')
    movies['GenreCorrected']=movies['GenreCorrected'].str.replace('true crime','crime')
    movies['GenreCorrected']=movies['GenreCorrected'].str.replace('\|007','')
    movies['GenreCorrected']=movies['GenreCorrected'].str.replace('slice of life','slice_of_life')
    movies['GenreCorrected']=movies['GenreCorrected'].str.replace('computer animation','animation')
    movies['GenreCorrected']=movies['GenreCorrected'].str.replace('gun fu','martial_arts')
    movies['GenreCorrected']=movies['GenreCorrected'].str.replace('j-horror','horror')
    movies['GenreCorrected']=movies['GenreCorrected'].str.replace(' \(shogi|chess\)','')
    movies['GenreCorrected']=movies['GenreCorrected'].str.replace('afghan war drama','war drama')
    movies['GenreCorrected']=movies['GenreCorrected'].str.replace('\|6 separate stories','')
    movies['GenreCorrected']=movies['GenreCorrected'].str.replace(' \(30min\)','')
    movies['GenreCorrected']=movies['GenreCorrected'].str.replace(' (road bicycle racing)','')
    movies['GenreCorrected']=movies['GenreCorrected'].str.replace(' v-cinema','')
    movies['GenreCorrected']=movies['GenreCorrected'].str.replace('tv miniseries','tv_miniseries')
    movies['GenreCorrected']=movies['GenreCorrected'].str.replace('\|docudrama','\|documentary|drama')
    movies['GenreCorrected']=movies['GenreCorrected'].str.replace(' in animation','|animation')
    movies['GenreCorrected']=movies['GenreCorrected'].str.replace('((adaptation).+)','')
    movies['GenreCorrected']=movies['GenreCorrected'].str.replace('((adaptated).+)','')
    movies['GenreCorrected']=movies['GenreCorrected'].str.replace('((adapted).+)','')
    movies['GenreCorrected']=movies['GenreCorrected'].str.replace('(( on ).+)','')
    movies['GenreCorrected']=movies['GenreCorrected'].str.replace('american football','sports')
    movies['GenreCorrected']=movies['GenreCorrected'].str.replace('dev\|nusrat jahan','sports')
    movies['GenreCorrected']=movies['GenreCorrected'].str.replace('television miniseries','tv_miniseries')
    movies['GenreCorrected']=movies['GenreCorrected'].str.replace(' \(artistic\)','')
    movies['GenreCorrected']=movies['GenreCorrected'].str.replace(' \|direct-to-dvd','')
    movies['GenreCorrected']=movies['GenreCorrected'].str.replace('history dram','history drama')
    movies['GenreCorrected']=movies['GenreCorrected'].str.replace('martial art','martial_arts')
    movies['GenreCorrected']=movies['GenreCorrected'].str.replace('psycho thriller,','psycho thriller')
    movies['GenreCorrected']=movies['GenreCorrected'].str.replace('\|1 girl\|3 suitors','')
    movies['GenreCorrected']=movies['GenreCorrected'].str.replace(' \(road bicycle racing\)','')
    filterE = movies['GenreCorrected']=="ero"
    movies.loc[filterE,'GenreCorrected']="adult"
    filterE = movies['GenreCorrected']=="music"
    movies.loc[filterE,'GenreCorrected']="musical"
    filterE = movies['GenreCorrected']=="-"
    movies.loc[filterE,'GenreCorrected']=''
    filterE = movies['GenreCorrected']=="comedy???drama"
    movies.loc[filterE,'GenreCorrected'] = "comedy|drama"
    filterE = movies['GenreCorrected']=="comedy???horror"
    movies.loc[filterE,'GenreCorrected'] = "comedy|horror"
    movies['GenreCorrected']=movies['GenreCorrected'].str.replace(' ','|')
    movies['GenreCorrected']=movies['GenreCorrected'].str.replace(',','|')
    movies['GenreCorrected']=movies['GenreCorrected'].str.replace('-','')
    movies['GenreCorrected']=movies['GenreCorrected'].str.replace('actionadventure','action|adventure')
    movies['GenreCorrected']=movies['GenreCorrected'].str.replace('actioncomedy','action|comedy')
    movies['GenreCorrected']=movies['GenreCorrected'].str.replace('actiondrama','action|drama')
    movies['GenreCorrected']=movies['GenreCorrected'].str.replace('actionlove','action|love')
    movies['GenreCorrected']=movies['GenreCorrected'].str.replace('actionmasala','action|masala')
    movies['GenreCorrected']=movies['GenreCorrected'].str.replace('actionchildren','action|children')

    movies['GenreCorrected']=movies['GenreCorrected'].str.replace('fantasychildren\|','fantasy|children')
    movies['GenreCorrected']=movies['GenreCorrected'].str.replace('fantasycomedy','fantasy|comedy')
    movies['GenreCorrected']=movies['GenreCorrected'].str.replace('fantasyperiod','fantasy|period')
    movies['GenreCorrected']=movies['GenreCorrected'].str.replace('cbctv_miniseries','tv_miniseries')
    movies['GenreCorrected']=movies['GenreCorrected'].str.replace('dramacomedy','drama|comedy')
    movies['GenreCorrected']=movies['GenreCorrected'].str.replace('dramacomedysocial','drama|comedy|social')
    movies['GenreCorrected']=movies['GenreCorrected'].str.replace('dramathriller','drama|thriller')
    movies['GenreCorrected']=movies['GenreCorrected'].str.replace('comedydrama','comedy|drama')
    movies['GenreCorrected']=movies['GenreCorrected'].str.replace('dramathriller','drama|thriller')
    movies['GenreCorrected']=movies['GenreCorrected'].str.replace('comedyhorror','comedy|horror')
    movies['GenreCorrected']=movies['GenreCorrected'].str.replace('sciencefiction','science_fiction')
    movies['GenreCorrected']=movies['GenreCorrected'].str.replace('adventurecomedy','adventure|comedy')
    movies['GenreCorrected']=movies['GenreCorrected'].str.replace('animationdrama','animation|drama')
    movies['GenreCorrected']=movies['GenreCorrected'].str.replace('\|\|','|')
    movies['GenreCorrected']=movies['GenreCorrected'].str.replace('muslim','religious')
    movies['GenreCorrected']=movies['GenreCorrected'].str.replace('thriler','thriller')
    movies['GenreCorrected']=movies['GenreCorrected'].str.replace('crimethriller','crime|thriller')
    movies['GenreCorrected']=movies['GenreCorrected'].str.replace('fantay','fantasy')
    movies['GenreCorrected']=movies['GenreCorrected'].str.replace('actionthriller','action|thriller')
    movies['GenreCorrected']=movies['GenreCorrected'].str.replace('comedysocial','comedy|social')
    movies['GenreCorrected']=movies['GenreCorrected'].str.replace('martialarts','martial_arts')
    movies['GenreCorrected']=movies['GenreCorrected'].str.replace('\|\(children\|poker\|karuta\)','')
    movies['GenreCorrected']=movies['GenreCorrected'].str.replace('epichistory','epic|history')

    movies['GenreCorrected']=movies['GenreCorrected'].str.replace('erotica','adult')
    movies['GenreCorrected']=movies['GenreCorrected'].str.replace('erotic','adult')

    movies['GenreCorrected']=movies['GenreCorrected'].str.replace('((\|produced\|).+)','')
    movies['GenreCorrected']=movies['GenreCorrected'].str.replace('chanbara','chambara')
    movies['GenreCorrected']=movies['GenreCorrected'].str.replace('comedythriller','comedy|thriller')
    movies['GenreCorrected']=movies['GenreCorrected'].str.replace('biblical','religious')
    movies['GenreCorrected']=movies['GenreCorrected'].str.replace('biblical','religious')
    movies['GenreCorrected']=movies['GenreCorrected'].str.replace('colour\|yellow\|productions\|eros\|international','')
    movies['GenreCorrected']=movies['GenreCorrected'].str.replace('\|directtodvd','')
    movies['GenreCorrected']=movies['GenreCorrected'].str.replace('liveaction','live|action')
    movies['GenreCorrected']=movies['GenreCorrected'].str.replace('melodrama','drama')
    movies['GenreCorrected']=movies['GenreCorrected'].str.replace('superheroes','superheroe')
    movies['GenreCorrected']=movies['GenreCorrected'].str.replace('gangsterthriller','gangster|thriller')

    movies['GenreCorrected']=movies['GenreCorrected'].str.replace('heistcomedy','comedy')
    movies['GenreCorrected']=movies['GenreCorrected'].str.replace('heist','action')
    movies['GenreCorrected']=movies['GenreCorrected'].str.replace('historic','history')
    movies['GenreCorrected']=movies['GenreCorrected'].str.replace('historydisaster','history|disaster')
    movies['GenreCorrected']=movies['GenreCorrected'].str.replace('warcomedy','war|comedy')
    movies['GenreCorrected']=movies['GenreCorrected'].str.replace('westerncomedy','western|comedy')
    movies['GenreCorrected']=movies['GenreCorrected'].str.replace('ancientcostume','costume')
    movies['GenreCorrected']=movies['GenreCorrected'].str.replace('computeranimation','animation')
    movies['GenreCorrected']=movies['GenreCorrected'].str.replace('dramatic','drama')
    movies['GenreCorrected']=movies['GenreCorrected'].str.replace('familya','family')
    movies['GenreCorrected']=movies['GenreCorrected'].str.replace('familya','family')
    movies['GenreCorrected']=movies['GenreCorrected'].str.replace('dramedy','drama|comedy')
    movies['GenreCorrected']=movies['GenreCorrected'].str.replace('dramaa','drama')
    movies['GenreCorrected']=movies['GenreCorrected'].str.replace('famil\|','family')

    movies['GenreCorrected']=movies['GenreCorrected'].str.replace('superheroe','superhero')
    movies['GenreCorrected']=movies['GenreCorrected'].str.replace('biogtaphy','biography')
    movies['GenreCorrected']=movies['GenreCorrected'].str.replace('devotionalbiography','devotional|biography')
    movies['GenreCorrected']=movies['GenreCorrected'].str.replace('docufiction','documentary|fiction')

    movies['GenreCorrected']=movies['GenreCorrected'].str.replace('familydrama','family|drama')

    movies['GenreCorrected']=movies['GenreCorrected'].str.replace('espionage','spy')
    movies['GenreCorrected']=movies['GenreCorrected'].str.replace('supeheroes','superhero')
    movies['GenreCorrected']=movies['GenreCorrected'].str.replace('romancefiction','romance|fiction')
    movies['GenreCorrected']=movies['GenreCorrected'].str.replace('horrorthriller','horror|thriller')

    movies['GenreCorrected']=movies['GenreCorrected'].str.replace('suspensethriller','suspense|thriller')
    movies['GenreCorrected']=movies['GenreCorrected'].str.replace('musicaliography','musical|biography')
    movies['GenreCorrected']=movies['GenreCorrected'].str.replace('triller','thriller')

    movies['GenreCorrected']=movies['GenreCorrected'].str.replace('\|\(fiction\)','|fiction')

    movies['GenreCorrected']=movies['GenreCorrected'].str.replace('romanceaction','romance|action')
    movies['GenreCorrected']=movies['GenreCorrected'].str.replace('romancecomedy','romance|comedy')
    movies['GenreCorrected']=movies['GenreCorrected'].str.replace('romancehorror','romance|horror')

    movies['GenreCorrected']=movies['GenreCorrected'].str.replace('romcom','romance|comedy')
    movies['GenreCorrected']=movies['GenreCorrected'].str.replace('rom\|com','romance|comedy')
    movies['GenreCorrected']=movies['GenreCorrected'].str.replace('satirical','satire')

    movies['GenreCorrected']=movies['GenreCorrected'].str.replace('science_fictionchildren','science_fiction|children')
    movies['GenreCorrected']=movies['GenreCorrected'].str.replace('homosexual','adult')
    movies['GenreCorrected']=movies['GenreCorrected'].str.replace('sexual','adult')

    movies['GenreCorrected']=movies['GenreCorrected'].str.replace('mockumentary','documentary')
    movies['GenreCorrected']=movies['GenreCorrected'].str.replace('periodic','period')
    movies['GenreCorrected']=movies['GenreCorrected'].str.replace('romanctic','romantic')
    movies['GenreCorrected']=movies['GenreCorrected'].str.replace('politics','political')
    movies['GenreCorrected']=movies['GenreCorrected'].str.replace('samurai','martial_arts')
    movies['GenreCorrected']=movies['GenreCorrected'].str.replace('tv_miniseries','series')
    movies['GenreCorrected']=movies['GenreCorrected'].str.replace('serial','series')

    filterE = movies['GenreCorrected']=="musical???comedy"
    movies.loc[filterE,'GenreCorrected'] = "musical|comedy"

    filterE = movies['GenreCorrected']=="roman|porno"
    movies.loc[filterE,'GenreCorrected'] = "adult"


    filterE = movies['GenreCorrected']=="action???masala"
    movies.loc[filterE,'GenreCorrected'] = "action|masala"


    filterE = movies['GenreCorrected']=="horror???thriller"
    movies.loc[filterE,'GenreCorrected'] = "horror|thriller"

    movies['GenreCorrected']=movies['GenreCorrected'].str.replace('family','children')
    movies['GenreCorrected']=movies['GenreCorrected'].str.replace('martial_arts','action')
    movies['GenreCorrected']=movies['GenreCorrected'].str.replace('horror','thriller')
    movies['GenreCorrected']=movies['GenreCorrected'].str.replace('war','action')
    movies['GenreCorrected']=movies['GenreCorrected'].str.replace('adventure','action')
    movies['GenreCorrected']=movies['GenreCorrected'].str.replace('science_fiction','action')
    movies['GenreCorrected']=movies['GenreCorrected'].str.replace('western','action')
    movies['GenreCorrected']=movies['GenreCorrected'].str.replace('western','action')
    movies['GenreCorrected']=movies['GenreCorrected'].str.replace('noir','black')
    movies['GenreCorrected']=movies['GenreCorrected'].str.replace('spy','action')
    movies['GenreCorrected']=movies['GenreCorrected'].str.replace('superhero','action')
    movies['GenreCorrected']=movies['GenreCorrected'].str.replace('social','')
    movies['GenreCorrected']=movies['GenreCorrected'].str.replace('suspense','action')


    filterE = movies['GenreCorrected']=="drama|romance|adult|children"
    movies.loc[filterE,'GenreCorrected'] = "drama|romance|adult"

    movies['GenreCorrected']=movies['GenreCorrected'].str.replace('\|???\|','|')
    movies['GenreCorrected']=movies['GenreCorrected'].str.strip(to_strip='\|')
    movies['GenreCorrected']=movies['GenreCorrected'].str.replace('actionner','action')
    movies['GenreCorrected']=movies['GenreCorrected'].str.strip()

    dataset_harmonized = movies.drop(columns=['Genre'])
    dataset_harmonized.rename(columns={ "GenreCorrected": "Genre"}, inplace=True)

    return dataset_harmonized
