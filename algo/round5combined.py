import json
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
from typing import Any

from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List
import string
import jsonpickle
import numpy as np
import statistics
import math

class Trader:
    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        # Orders to be placed on exchange matching engine
        result = {'AMETHYSTS' : [], 'STARFRUIT' : [], 'ORCHIDS' : [], 'CHOCOLATE' : [], 'STRAWBERRIES' : [], 'ROSES' : [], 'GIFT_BASKET' : [], 'COCONUT' : [], 'COCONUT_COUPON' : []}

        if state.traderData == '':
            mydata = {}
            mydata['position_limit'] = {'AMETHYSTS' : 20, 'STARFRUIT' : 20, 'ORCHIDS' : 100, 'CHOCOLATE' : 250, 'STRAWBERRIES' : 350, 'ROSES' : 60, "GIFT_BASKET" : 60, "COCONUT":300, "COCONUT_COUPON":600}
            # inputs used to estimate starfruit price
            mydata['starfruit_cache'] = []
            mydata['starfruit_dim'] = 5
            mydata['Rhianna_flag'] = -1 #trader bot flag
            # manually inputing the cdf table as I can only use limited python libraries
            mydata['cdf_table'] = {-3.0: 0.001349898031630093, -2.99: 0.0013948872354922468, -2.98: 0.0014412419173400136, -2.97: 0.0014889987452374632, -2.96: 0.0015381952117380577, -2.95: 0.0015888696473648667, -2.94: 0.0016410612341569964, -2.93: 0.0016948100192772598, -2.92: 0.0017501569286760992, -2.91: 0.0018071437808064273, -2.9: 0.0018658133003840375, -2.89: 0.0019262091321878587, -2.88: 0.0019883758548943256, -2.87: 0.0020523589949397536, -2.86: 0.0021182050404046217, -2.85: 0.0021859614549132405, -2.84: 0.0022556766915423202, -2.83: 0.0023274002067315515, -2.82: 0.0024011824741892525, -2.81: 0.002477074998785861, -2.8: 0.002555130330427932, -2.79: 0.0026354020779049527, -2.78: 0.0027179449227012578, -2.77: 0.002802814632765025, -2.76: 0.0028900680762261473, -2.75: 0.0029797632350545547, -2.74: 0.003071959218650488, -2.73: 0.003166716277357793, -2.72: 0.003264095815891309, -2.71: 0.0033641604066691933, -2.7: 0.0034669738030406647, -2.69: 0.0035726009523997368, -2.68: 0.00368110800917498, -2.67: 0.0037925623476854887, -2.66: 0.003907032574852774, -2.65: 0.004024588542758306, -2.64: 0.004145301361036037, -2.63: 0.004269243409089351, -2.62: 0.004396488348121309, -2.61: 0.004527111132967324, -2.6: 0.004661188023718747, -2.59: 0.004798796597126178, -2.58: 0.004940015757770644, -2.57: 0.0050849257489910355, -2.56: 0.0052336081635557825, -2.55: 0.005386145954066687, -2.54: 0.005542623443082601, -2.53: 0.005703126332950699, -2.52: 0.005867741715332562, -2.51: 0.006036558080412664, -2.5: 0.006209665325776132, -2.49: 0.006387154764943171, -2.48: 0.006569119135546763, -2.47: 0.006755652607140648, -2.46: 0.006946850788624312, -2.45: 0.007142810735271415, -2.44: 0.007343630955348342, -2.43: 0.007549411416309199, -2.42: 0.0077602535505536425, -2.41: 0.007976260260733725, -2.4: 0.008197535924596131, -2.39: 0.008424186399345683, -2.38: 0.008656319025516541, -2.37: 0.008894042630336772, -2.36: 0.00913746753057267, -2.35: 0.009386705534838566, -2.34: 0.00964186994535833, -2.33: 0.009903075559164245, -2.32: 0.01017043866871968, -2.31: 0.010444077061951081, -2.3: 0.010724110021675809, -2.29: 0.011010658324411388, -2.28: 0.01130384423855279, -2.27: 0.011603791521903536, -2.26: 0.011910625418547066, -2.25: 0.012224472655044696, -2.24: 0.012545461435946563, -2.23: 0.012873721438602014, -2.22: 0.01320938380725627, -2.21: 0.013552581146419983, -2.2: 0.013903447513498595, -2.19: 0.014262118410668878, -2.18: 0.014628730775989246, -2.17: 0.015003422973732207, -2.16: 0.015386334783925445, -2.15: 0.0157776073910905, -2.14: 0.01617738337216608, -2.13: 0.01658580668360501, -2.12: 0.017003022647632787, -2.11: 0.017429177937657088, -2.1: 0.017864420562816542, -2.09: 0.01830889985165897, -2.08: 0.01876276643493775, -2.07: 0.019226172227517282, -2.06: 0.019699270409376895, -2.05: 0.020182215405704394, -2.04: 0.020675162866070042, -2.03: 0.02117826964267227, -2.02: 0.021691693767646774, -2.01: 0.022215594429431475, -2.0: 0.0227501319481792, -1.99: 0.02329546775021182, -1.98: 0.023851764341508513, -1.97: 0.024419185280222543, -1.96: 0.024997895148220435, -1.95: 0.02558805952163861, -1.94: 0.026189844940452685, -1.93: 0.02680341887705495, -1.92: 0.027428949703836802, -1.91: 0.0280666066597725, -1.9: 0.028716559816001803, -1.89: 0.02937898004040943, -1.88: 0.03005403896119979, -1.87: 0.030741908929465954, -1.86: 0.0314427629807527, -1.85: 0.03215677479561371, -1.84: 0.03288411865916389, -1.83: 0.0336249694196283, -1.82: 0.03437950244588998, -1.81: 0.03514789358403879, -1.8: 0.03593031911292579, -1.79: 0.03672695569872628, -1.78: 0.03753798034851679, -1.77: 0.03836357036287122, -1.76: 0.03920390328748265, -1.75: 0.040059156863817086, -1.74: 0.040929508978807365, -1.73: 0.04181513761359493, -1.72: 0.04271622079132892, -1.71: 0.0436329365240319, -1.7: 0.04456546275854304, -1.69: 0.045513977321549805, -1.68: 0.046478657863720046, -1.67: 0.04745968180294733, -1.66: 0.04845722626672282, -1.65: 0.0494714680336481, -1.64: 0.05050258347410371, -1.63: 0.051550748490089365, -1.62: 0.05261613845425206, -1.61: 0.053698928148119655, -1.6: 0.054799291699557974, -1.59: 0.05591740251946941, -1.58: 0.057053433237754185, -1.57: 0.05820755563855299, -1.56: 0.05937994059479304, -1.55: 0.06057075800205901, -1.54: 0.061780176711811886, -1.53: 0.06300836446397844, -1.52: 0.06425548781893582, -1.51: 0.0655217120889165, -1.5: 0.06680720126885806, -1.49: 0.06811211796672544, -1.48: 0.06943662333333173, -1.47: 0.07078087699168552, -1.46: 0.07214503696589378, -1.45: 0.07352925960964836, -1.44: 0.07493369953432708, -1.43: 0.07635850953673913, -1.42: 0.07780384052654638, -1.41: 0.07926984145339239, -1.4: 0.08075665923377107, -1.39: 0.08226443867766892, -1.38: 0.08379332241501436, -1.37: 0.08534345082196698, -1.36: 0.08691496194708503, -1.35: 0.08850799143740196, -1.34: 0.09012267246445244, -1.33: 0.09175913565028077, -1.32: 0.09341750899347179, -1.31: 0.09509791779523902, -1.3: 0.09680048458561036, -1.29: 0.09852532904974787, -1.28: 0.10027256795444206, -1.27: 0.10204231507481915, -1.26: 0.1038346811213004, -1.25: 0.10564977366685535, -1.24: 0.10748769707458694, -1.23: 0.10934855242569191, -1.22: 0.11123243744783456, -1.21: 0.11313944644397739, -1.2: 0.11506967022170822, -1.19: 0.11702319602310873, -1.18: 0.11900010745520073, -1.17: 0.12100048442101824, -1.16: 0.12302440305134343, -1.15: 0.12507193563715036, -1.14: 0.1271431505627983, -1.13: 0.12923811224001786, -1.12: 0.1313568810427307, -1.11: 0.13349951324274717, -1.1: 0.13566606094638267, -1.09: 0.1378565720320355, -1.08: 0.140071090088769, -1.07: 0.14230965435593923, -1.06: 0.1445722996639096, -1.05: 0.1468590563758959, -1.04: 0.1491699503309814, -1.03: 0.15150500278834367, -1.02: 0.15386423037273478, -1.01: 0.15624764502125466, -1.0: 0.15865525393145707, -0.99: 0.1610870595108309, -0.98: 0.16354305932769236, -0.97: 0.16602324606352958, -0.96: 0.16852760746683781, -0.95: 0.17105612630848183, -0.94: 0.1736087803386246, -0.93: 0.1761855422452579, -0.92: 0.17878637961437172, -0.91: 0.1814112548917972, -0.9: 0.18406012534675947, -0.89: 0.18673294303717264, -0.88: 0.18942965477671214, -0.87: 0.19215020210369615, -0.86: 0.1948945212518084, -0.85: 0.19766254312269238, -0.84: 0.20045419326044966, -0.83: 0.2032693918280684, -0.82: 0.20610805358581302, -0.81: 0.2089700878716016, -0.8: 0.2118553985833967, -0.79: 0.21476388416363712, -0.78: 0.21769543758573312, -0.77: 0.22064994634264962, -0.76: 0.22362729243759938, -0.75: 0.2266273523768682, -0.74: 0.22964999716479056, -0.73: 0.2326950923008974, -0.72: 0.23576249777925118, -0.71: 0.2388520680899867, -0.7: 0.24196365222307303, -0.69: 0.24509709367430943, -0.68: 0.24825223045357053, -0.67: 0.25142889509531013, -0.66: 0.25462691467133614, -0.65: 0.2578461108058647, -0.64: 0.26108629969286157, -0.63: 0.26434729211567753, -0.62: 0.26762889346898305, -0.61: 0.27093090378300566, -0.6: 0.2742531177500736, -0.59: 0.27759532475346493, -0.58: 0.28095730889856435, -0.57: 0.28433884904632417, -0.56: 0.28773971884902705, -0.55: 0.29115968678834636, -0.54: 0.294598516215698, -0.53: 0.29805596539487644, -0.52: 0.3015317875469662, -0.51: 0.3050257308975194, -0.5: 0.3085375387259869, -0.49: 0.31206694941739055, -0.48: 0.31561369651622256, -0.47: 0.3191775087825558, -0.46: 0.32275811025034773, -0.45: 0.32635522028791997, -0.44: 0.32996855366059363, -0.43: 0.3335978205954577, -0.42: 0.3372427268482495, -0.41: 0.3409029737723226, -0.4: 0.3445782583896758, -0.39: 0.3482682734640176, -0.38: 0.3519727075758372, -0.37: 0.3556912451994533, -0.36: 0.35942356678200876, -0.35: 0.3631693488243809, -0.34: 0.36692826396397193, -0.33: 0.37069998105934643, -0.32: 0.37448416527667994, -0.31: 0.3782804781779807, -0.3: 0.3820885778110474, -0.29: 0.3859081188011227, -0.28: 0.3897387524442028, -0.27: 0.3935801268019605, -0.26: 0.3974318867982395, -0.25: 0.4012936743170763, -0.24: 0.40516512830220414, -0.23: 0.40904588485799415, -0.22: 0.41293557735178543, -0.21: 0.4168338365175577, -0.2: 0.42074029056089696, -0.19: 0.42465456526520456, -0.18: 0.42857628409909926, -0.17: 0.4325050683249616, -0.16: 0.4364405371085672, -0.15: 0.4403823076297575, -0.14: 0.44432999519409355, -0.13: 0.44828321334543886, -0.12: 0.45224157397941617, -0.11: 0.4562046874576832, -0.1: 0.460172162722971, -0.09: 0.4641436074148279, -0.08: 0.4681186279860126, -0.07: 0.47209682981947887, -0.06: 0.47607781734589316, -0.05: 0.4800611941616275, -0.04: 0.48404656314716926, -0.03: 0.48803352658588733, -0.02: 0.492021686283098, -0.01: 0.4960106436853684, 0.0: 0.5, 0.01: 0.5039893563146316, 0.02: 0.5079783137169019, 0.03: 0.5119664734141126, 0.04: 0.5159534368528308, 0.05: 0.5199388058383725, 0.06: 0.5239221826541068, 0.07: 0.5279031701805211, 0.08: 0.5318813720139874, 0.09: 0.5358563925851721, 0.1: 0.539827837277029, 0.11: 0.5437953125423168, 0.12: 0.5477584260205839, 0.13: 0.5517167866545611, 0.14: 0.5556700048059064, 0.15: 0.5596176923702425, 0.16: 0.5635594628914329, 0.17: 0.5674949316750384, 0.18: 0.5714237159009007, 0.19: 0.5753454347347955, 0.2: 0.579259709439103, 0.21: 0.5831661634824423, 0.22: 0.5870644226482146, 0.23: 0.5909541151420059, 0.24: 0.5948348716977958, 0.25: 0.5987063256829237, 0.26: 0.6025681132017605, 0.27: 0.6064198731980396, 0.28: 0.6102612475557972, 0.29: 0.6140918811988774, 0.3: 0.6179114221889526, 0.31: 0.6217195218220193, 0.32: 0.6255158347233201, 0.33: 0.6293000189406536, 0.34: 0.6330717360360281, 0.35: 0.6368306511756191, 0.36: 0.6405764332179913, 0.37: 0.6443087548005467, 0.38: 0.6480272924241628, 0.39: 0.6517317265359823, 0.4: 0.6554217416103242, 0.41: 0.6590970262276774, 0.42: 0.6627572731517505, 0.43: 0.6664021794045423, 0.44: 0.6700314463394064, 0.45: 0.67364477971208, 0.46: 0.6772418897496523, 0.47: 0.6808224912174442, 0.48: 0.6843863034837774, 0.49: 0.6879330505826095, 0.5: 0.6914624612740131, 0.51: 0.6949742691024806, 0.52: 0.6984682124530338, 0.53: 0.7019440346051236, 0.54: 0.705401483784302, 0.55: 0.7088403132116536, 0.56: 0.712260281150973, 0.57: 0.7156611509536759, 0.58: 0.7190426911014356, 0.59: 0.7224046752465351, 0.6: 0.7257468822499265, 0.61: 0.7290690962169943, 0.62: 0.732371106531017, 0.63: 0.7356527078843225, 0.64: 0.7389137003071384, 0.65: 0.7421538891941353, 0.66: 0.7453730853286639, 0.67: 0.7485711049046899, 0.68: 0.7517477695464294, 0.69: 0.7549029063256906, 0.7: 0.758036347776927, 0.71: 0.7611479319100133, 0.72: 0.7642375022207488, 0.73: 0.7673049076991025, 0.74: 0.7703500028352095, 0.75: 0.7733726476231317, 0.76: 0.7763727075624006, 0.77: 0.7793500536573503, 0.78: 0.7823045624142668, 0.79: 0.7852361158363629, 0.8: 0.7881446014166034, 0.81: 0.7910299121283983, 0.82: 0.7938919464141869, 0.83: 0.7967306081719316, 0.84: 0.7995458067395503, 0.85: 0.8023374568773076, 0.86: 0.8051054787481916, 0.87: 0.8078497978963038, 0.88: 0.8105703452232879, 0.89: 0.8132670569628273, 0.9: 0.8159398746532405, 0.91: 0.8185887451082028, 0.92: 0.8212136203856283, 0.93: 0.8238144577547422, 0.94: 0.8263912196613754, 0.95: 0.8289438736915182, 0.96: 0.8314723925331622, 0.97: 0.8339767539364704, 0.98: 0.8364569406723077, 0.99: 0.8389129404891691, 1.0: 0.8413447460685429, 1.01: 0.8437523549787453, 1.02: 0.8461357696272652, 1.03: 0.8484949972116563, 1.04: 0.8508300496690187, 1.05: 0.8531409436241041, 1.06: 0.8554277003360904, 1.07: 0.8576903456440608, 1.08: 0.859928909911231, 1.09: 0.8621434279679645, 1.1: 0.8643339390536173, 1.11: 0.8665004867572528, 1.12: 0.8686431189572693, 1.13: 0.8707618877599821, 1.14: 0.8728568494372018, 1.15: 0.8749280643628496, 1.16: 0.8769755969486566, 1.17: 0.8789995155789818, 1.18: 0.8809998925447993, 1.19: 0.8829768039768913, 1.2: 0.8849303297782918, 1.21: 0.8868605535560226, 1.22: 0.8887675625521654, 1.23: 0.8906514475743081, 1.24: 0.8925123029254131, 1.25: 0.8943502263331446, 1.26: 0.8961653188786995, 1.27: 0.8979576849251809, 1.28: 0.8997274320455579, 1.29: 0.9014746709502521, 1.3: 0.9031995154143897, 1.31: 0.904902082204761, 1.32: 0.9065824910065282, 1.33: 0.9082408643497193, 1.34: 0.9098773275355476, 1.35: 0.911492008562598, 1.36: 0.913085038052915, 1.37: 0.914656549178033, 1.38: 0.9162066775849856, 1.39: 0.917735561322331, 1.4: 0.9192433407662289, 1.41: 0.9207301585466077, 1.42: 0.9221961594734536, 1.43: 0.9236414904632608, 1.44: 0.925066300465673, 1.45: 0.9264707403903516, 1.46: 0.9278549630341062, 1.47: 0.9292191230083144, 1.48: 0.9305633766666683, 1.49: 0.9318878820332746, 1.5: 0.9331927987311419, 1.51: 0.9344782879110836, 1.52: 0.9357445121810641, 1.53: 0.9369916355360216, 1.54: 0.9382198232881881, 1.55: 0.939429241997941, 1.56: 0.940620059405207, 1.57: 0.941792444361447, 1.58: 0.9429465667622459, 1.59: 0.9440825974805306, 1.6: 0.945200708300442, 1.61: 0.9463010718518804, 1.62: 0.9473838615457479, 1.63: 0.9484492515099107, 1.64: 0.9494974165258963, 1.65: 0.9505285319663519, 1.66: 0.9515427737332772, 1.67: 0.9525403181970526, 1.68: 0.9535213421362799, 1.69: 0.9544860226784502, 1.7: 0.955434537241457, 1.71: 0.9563670634759681, 1.72: 0.957283779208671, 1.73: 0.9581848623864051, 1.74: 0.9590704910211927, 1.75: 0.9599408431361829, 1.76: 0.9607960967125173, 1.77: 0.9616364296371288, 1.78: 0.9624620196514833, 1.79: 0.9632730443012737, 1.8: 0.9640696808870742, 1.81: 0.9648521064159612, 1.82: 0.9656204975541101, 1.83: 0.9663750305803717, 1.84: 0.9671158813408361, 1.85: 0.9678432252043863, 1.86: 0.9685572370192473, 1.87: 0.9692580910705341, 1.88: 0.9699459610388002, 1.89: 0.9706210199595906, 1.9: 0.9712834401839981, 1.91: 0.9719333933402275, 1.92: 0.9725710502961632, 1.93: 0.973196581122945, 1.94: 0.9738101550595473, 1.95: 0.9744119404783614, 1.96: 0.9750021048517795, 1.97: 0.9755808147197774, 1.98: 0.9761482356584915, 1.99: 0.9767045322497881, 2.0: 0.9772498680518208, 2.01: 0.9777844055705686, 2.02: 0.9783083062323532, 2.03: 0.9788217303573278, 2.04: 0.9793248371339299, 2.05: 0.9798177845942956, 2.06: 0.9803007295906231, 2.07: 0.9807738277724827, 2.08: 0.9812372335650622, 2.09: 0.981691100148341, 2.1: 0.9821355794371834, 2.11: 0.9825708220623429, 2.12: 0.9829969773523672, 2.13: 0.983414193316395, 2.14: 0.9838226166278339, 2.15: 0.9842223926089095, 2.16: 0.9846136652160745, 2.17: 0.9849965770262678, 2.18: 0.9853712692240107, 2.19: 0.9857378815893312, 2.2: 0.9860965524865014, 2.21: 0.98644741885358, 2.22: 0.9867906161927438, 2.23: 0.987126278561398, 2.24: 0.9874545385640534, 2.25: 0.9877755273449553, 2.26: 0.988089374581453, 2.27: 0.9883962084780965, 2.28: 0.9886961557614472, 2.29: 0.9889893416755886, 2.3: 0.9892758899783242, 2.31: 0.989555922938049, 2.32: 0.9898295613312803, 2.33: 0.9900969244408357, 2.34: 0.9903581300546417, 2.35: 0.9906132944651614, 2.36: 0.9908625324694273, 2.37: 0.9911059573696632, 2.38: 0.9913436809744834, 2.39: 0.9915758136006543, 2.4: 0.9918024640754038, 2.41: 0.9920237397392663, 2.42: 0.9922397464494463, 2.43: 0.9924505885836908, 2.44: 0.9926563690446517, 2.45: 0.9928571892647285, 2.46: 0.9930531492113757, 2.47: 0.9932443473928594, 2.48: 0.9934308808644532, 2.49: 0.9936128452350568, 2.5: 0.9937903346742238, 2.51: 0.9939634419195873, 2.52: 0.9941322582846674, 2.53: 0.9942968736670493, 2.54: 0.9944573765569173, 2.55: 0.9946138540459333, 2.56: 0.9947663918364442, 2.57: 0.994915074251009, 2.58: 0.9950599842422294, 2.59: 0.9952012034028738, 2.6: 0.9953388119762813, 2.61: 0.9954728888670327, 2.62: 0.9956035116518787, 2.63: 0.9957307565909107, 2.64: 0.9958546986389639, 2.65: 0.9959754114572417, 2.66: 0.9960929674251472, 2.67: 0.9962074376523146, 2.68: 0.996318891990825, 2.69: 0.9964273990476002, 2.7: 0.9965330261969594, 2.71: 0.9966358395933308, 2.72: 0.9967359041841087, 2.73: 0.9968332837226422, 2.74: 0.9969280407813496, 2.75: 0.9970202367649454, 2.76: 0.9971099319237738, 2.77: 0.997197185367235, 2.78: 0.9972820550772987, 2.79: 0.9973645979220951, 2.8: 0.997444869669572, 2.81: 0.9975229250012141, 2.82: 0.9975988175258107, 2.83: 0.9976725997932685, 2.84: 0.9977443233084576, 2.85: 0.9978140385450868, 2.86: 0.9978817949595954, 2.87: 0.9979476410050603, 2.88: 0.9980116241451057, 2.89: 0.9980737908678121, 2.9: 0.998134186699616, 2.91: 0.9981928562191936, 2.92: 0.9982498430713239, 2.93: 0.9983051899807227, 2.94: 0.998358938765843, 2.95: 0.9984111303526352, 2.96: 0.998461804788262, 2.97: 0.9985110012547626, 2.98: 0.99855875808266, 2.99: 0.9986051127645078, 3.0: 0.9986501019683699}

            # have to use this library to store data
            traderData = jsonpickle.encode(mydata)
            mydata = jsonpickle.decode(traderData)
        else:
            mydata = jsonpickle.decode(state.traderData)

        # compute order for Amethysts, price fluctuates around 10000
        result['AMETHYSTS'] = self.compute_orders_amethysts(state, acc_bid = 10000, acc_ask = 10000, POSITION_LIMIT = mydata['position_limit'])        
        
        # trading Starfruit, get current orders in the market
        buy_orders = list(state.order_depths['STARFRUIT'].buy_orders.items())
        best_buy_pr, buy_vol = buy_orders[0]
        sell_orders = list(state.order_depths['STARFRUIT'].sell_orders.items())
        best_sell_pr, sell_vol = sell_orders[0]

        # cache starfruit mid price
        mydata['starfruit_cache'].append((best_buy_pr+best_sell_pr)/2.0)
        nxt_SF_price = 0
        # moving average, if we have enough historic prices, use below to forecast next price
        if len(mydata['starfruit_cache']) > mydata['starfruit_dim']:
            mydata['starfruit_cache'].pop(0)
            nxt_SF_price = int(round(sum(mydata['starfruit_cache'])/float(len(mydata['starfruit_cache']))))
            result['STARFRUIT'] = self.compute_orders_starfruit(state, acc_bid = int(round(nxt_SF_price-1,0)), acc_ask = math.floor(nxt_SF_price+1), POSITION_LIMIT = mydata['position_limit'])

        # compute order for Orchids
        result['ORCHIDS'], conversions = self.compute_orders_orchids_new(state, mydata)

        # compute orders for basket and contents
        orders = self.compute_orders_basket(state.order_depths, state = state, POSITION_LIMIT = mydata['position_limit'], mydata = mydata)
        result['GIFT_BASKET'] += orders['GIFT_BASKET'] 
        # did not find a stable profiting strategy for strawberries, not trading
        #result['STRAWBERRIES'] += orders['STRAWBERRIES']
        result['CHOCOLATE'] += orders['CHOCOLATE']
        result['ROSES'] += orders['ROSES']
        
        # compute order for coconut and coconut coupon
        orders = self.compute_orders_ccnc(state, mydata)
        result['COCONUT'] = orders['COCONUT']
        result['COCONUT_COUPON'] = orders['COCONUT_COUPON']        
        
        traderData = jsonpickle.encode(mydata)
        logger.print(f";order;{result}")
        logger.flush(state, result, conversions, traderData)
        return result, conversions, traderData
    
    def compute_orders_amethysts(self, state: TradingState, acc_bid, acc_ask, POSITION_LIMIT):
        
        product = 'AMETHYSTS'
        orders: list[Order] = []

        # find buy orders
        buy_orders = list(state.order_depths[product].buy_orders.items())
        best_buy_pr, buy_vol = buy_orders[0]

        # find sell orders
        sell_orders = list(state.order_depths[product].sell_orders.items())
        best_sell_pr, sell_vol = sell_orders[0]

        pos_limt = POSITION_LIMIT[product]

        # get current position
        cpos= 0
        if product in state.position.keys():
            cpos = state.position[product]

        mx_with_buy = -1
        buy_cpos_update = cpos
        for ask, vol in sell_orders:
            # MATCHING ASK ORDERS
            # buy for less/equal to the price I am willing to pay depending on my position
            if ((ask < acc_bid) or ((cpos<0) and (ask == acc_bid))) and buy_cpos_update < pos_limt:
                mx_with_buy = max(mx_with_buy, ask)
                order_for = min(-vol, pos_limt - cpos)
                buy_cpos_update += order_for
                assert(order_for >= 0)
                orders.append(Order(product, ask, order_for))


        sell_cpos_update = cpos
        for bid, vol in buy_orders:
            # MATCHING BUY ORDERS
            # sell for more/equal to the price I am willing to sell
            if ((bid > acc_ask) or ((cpos>0) and (bid == acc_ask))) and sell_cpos_update > -pos_limt:
                order_for = max(-vol, -pos_limt-sell_cpos_update)
                sell_cpos_update += order_for
                assert(order_for <= 0)
                orders.append(Order(product, bid, order_for))

        
        # Market making
        undercut_buy = best_buy_pr + 1
        
        # depending on the current position list buy order beating the best buy price
        if (buy_cpos_update < pos_limt) and (cpos < 0): # short position
            num = min(40, pos_limt - buy_cpos_update)
            orders.append(Order(product, min(undercut_buy + 1, acc_bid-1), num))
            buy_cpos_update += num

        if (buy_cpos_update < pos_limt) and (cpos > 15): # long position
            num = min(40, pos_limt - buy_cpos_update)
            orders.append(Order(product, min(undercut_buy - 1, acc_bid-1), num))
            buy_cpos_update += num

        if buy_cpos_update < pos_limt: # position between 1 to 14
            num = min(40, pos_limt - buy_cpos_update)
            orders.append(Order(product, min(undercut_buy, acc_bid-1), num))
            buy_cpos_update += num

        undercut_sell = best_sell_pr - 1
        # depending on the current position list sell order beating the best sell price
        if (sell_cpos_update > -pos_limt) and (cpos > 0): # long position
            num = max(-40, -pos_limt-sell_cpos_update) # sell 40 or 
            orders.append(Order(product, max(undercut_sell-1, acc_ask+1), num))
            sell_cpos_update += num

        if (sell_cpos_update > -pos_limt) and (cpos < -15):
            num = max(-40, -pos_limt-sell_cpos_update)
            orders.append(Order(product, max(undercut_sell+1, acc_ask+1), num))
            sell_cpos_update += num

        if sell_cpos_update > -pos_limt:
            num = max(-40, -pos_limt-sell_cpos_update)
            orders.append(Order(product, max(undercut_sell, acc_ask+1), num))
            sell_cpos_update += num

        return orders
    
    def compute_orders_starfruit(self, state: TradingState, acc_bid, acc_ask, POSITION_LIMIT):
        # same trading strategy as amethysts
        product = 'STARFRUIT'
        
        orders: list[Order] = []

        buy_orders = list(state.order_depths[product].buy_orders.items())
        best_buy_pr, buy_vol = buy_orders[0]

        sell_orders = list(state.order_depths[product].sell_orders.items())
        best_sell_pr, sell_vol = sell_orders[0]

        pos_limt = POSITION_LIMIT[product]

        cpos= 0
        if product in state.position.keys():
            cpos = state.position[product]

        mx_with_buy = -1
        buy_cpos_update = cpos
        for ask, vol in sell_orders:
            if ((ask <= acc_bid) or ((buy_cpos_update<0) and (ask == acc_bid+1))) and buy_cpos_update < pos_limt:
                mx_with_buy = max(mx_with_buy, ask)
                order_for = min(-vol, pos_limt - cpos)
                buy_cpos_update += order_for
                assert(order_for >= 0)
                orders.append(Order(product, ask, order_for))

        cpos= 0
        if product in state.position.keys():
            cpos = state.position[product]
        sell_cpos_update = cpos
        for bid, vol in buy_orders:
            if ((bid >= acc_ask) or ((sell_cpos_update>0) and (bid+1 == acc_ask))) and sell_cpos_update > -pos_limt:
                order_for = max(-vol, -pos_limt-sell_cpos_update)
                sell_cpos_update += order_for
                assert(order_for <= 0)
                orders.append(Order(product, bid, order_for))
        
        # undercut
        if buy_cpos_update < pos_limt:
            num = pos_limt - buy_cpos_update
            orders.append(Order(product, min(best_buy_pr + 1, acc_bid), num))
            buy_cpos_update += num
        
        if sell_cpos_update > -pos_limt:
            num = -pos_limt-sell_cpos_update
            orders.append(Order(product, max(best_sell_pr - 1, acc_ask), num))
            sell_cpos_update += num

        return orders

    def compute_orders_orchids_new(self, state:TradingState, mydata):
        p = 'ORCHIDS'

        orders: list[Order] = []

        buy_orders = list(state.order_depths[p].buy_orders.items())
        best_buy_pr, buy_vol = buy_orders[0]

        sell_orders = list(state.order_depths[p].sell_orders.items())
        best_sell_pr, sell_vol = sell_orders[0]

        mid_price = (best_buy_pr + best_sell_pr)/2
        pos_limt = mydata['position_limit'][p]

        # trading with south island, realistic price is ask/bidprice + import/export + transport fee
        # there are also storage cost involved when buying orchids but converting every time tick yields best profit therefore it is not taken into account
        buy_from_south = state.observations.conversionObservations[p].askPrice + state.observations.conversionObservations[p].importTariff + state.observations.conversionObservations[p].transportFees        
        sell_to_south = state.observations.conversionObservations[p].bidPrice - state.observations.conversionObservations[p].exportTariff - state.observations.conversionObservations[p].transportFees

        sell_cpos = state.position.get(p,0)
        # I sell to north and buy from south, if north buys at higher price
        if (buy_from_south < best_buy_pr) and (sell_cpos > -pos_limt):
            for bid, vol in buy_orders:
                if buy_from_south < bid:
                    order_for = min(pos_limt + sell_cpos,vol)
                    orders.append(Order(p,bid,-order_for))
                    sell_cpos += -order_for

        buy_cpos = state.position.get(p,0)
        # I buy from north and sell to south, if north sells at lower price
        if (sell_to_south > best_sell_pr) and (buy_cpos < pos_limt):
            for ask, vol in sell_orders:
                if sell_to_south > ask:
                    order_for = min(pos_limt - buy_cpos,-vol)
                    orders.append(Order(p,ask,order_for))
                    buy_cpos += order_for

        # Market making
        # list sell order if I can make profit
        if (buy_from_south < mid_price) and (sell_cpos > -pos_limt):
            order_for = pos_limt + sell_cpos
            price = min(int(math.ceil(buy_from_south)), int(math.ceil(mid_price)))
            orders.append(Order(p,price,-order_for))
        # list buy order if I can make profit
        if (sell_to_south > mid_price) and (buy_cpos < pos_limt):
            order_for = pos_limt - buy_cpos
            price = max(int(math.floor(sell_to_south)), int(math.floor(mid_price)))
            orders.append(Order(p,price,order_for))

        #logger.print(f"buy_from_south:{buy_from_south},sell_to_south:{sell_to_south},midprice:{mid_price};{state.observations.conversionObservations[p].importTariff};{state.observations.conversionObservations[p].exportTariff};{state.observations.conversionObservations[p].transportFees}")

        # amount to trade with south island, I am going to convert every tick (arbitrage)
        conversions = -state.position.get(p,0)
        
        return orders, conversions

    def compute_orders_basket(self, order_depth, state: TradingState, POSITION_LIMIT, mydata):

        orders = {'STRAWBERRIES' : [], 'CHOCOLATE': [], 'ROSES' : [], 'GIFT_BASKET' : []}
        prods = ['STRAWBERRIES', 'CHOCOLATE', 'ROSES', 'GIFT_BASKET']
        osell, obuy, best_sell, best_buy, worst_sell, worst_buy, mid_price, vol_buy, vol_sell = {}, {}, {}, {}, {}, {}, {}, {}, {}

        # get orders for all products
        for p in prods:
            osell[p] = list(state.order_depths[p].sell_orders.items())
            obuy[p] = list(state.order_depths[p].buy_orders.items())
            best_buy[p] = 0
            if obuy[p]:
                best_buy[p], vol_buy[p] = obuy[p][0]
                worst_buy[p], dummy = obuy[p][-1]
            best_sell[p] = 0
            if osell[p]:
                best_sell[p], vol_sell[p] = osell[p][0]
                worst_sell[p], dummy = osell[p][-1]
            mid_price[p] = (best_sell[p] + best_buy[p])/2
        
        # basket contains 6 strawberries, 4 chocolates and 1 rose
        # therefore price for basket = 6*strawberry + 4*chocolate + 1*rose + premium
        # mean:379.4904833333333;median381.0;std76.42310842343252
        premium = 379.5
        res_buy = mid_price['GIFT_BASKET'] - mid_price['STRAWBERRIES']*6 - mid_price['CHOCOLATE']*4 - mid_price['ROSES'] - premium
        
        std = 76.4
        tot = mid_price['STRAWBERRIES']*6 + mid_price['CHOCOLATE']*4 + mid_price['ROSES']
        buy_cpos_gb = state.position.get('GIFT_BASKET',0)
        # gift basket underpriced, buy now and sell later
        if (res_buy < -std*0.5) and (res_buy > -std*2) and (buy_cpos_gb < POSITION_LIMIT['GIFT_BASKET']):
            for ask, vol in osell['GIFT_BASKET']:
                if (ask <= tot + premium) and (buy_cpos_gb < POSITION_LIMIT['GIFT_BASKET']):
                    order_for = min(POSITION_LIMIT['GIFT_BASKET'] - buy_cpos_gb,-vol)
                    orders['GIFT_BASKET'].append(Order('GIFT_BASKET',ask,order_for))
                    buy_cpos_gb += order_for
        
        sell_cpos_gb = state.position.get('GIFT_BASKET',0)
        # gift basket overpriced, sell now and buy later
        if (res_buy > std*0.5) and (res_buy < std*2.0)and (sell_cpos_gb > -POSITION_LIMIT['GIFT_BASKET']):
            for bid, vol in obuy['GIFT_BASKET']:
                if (bid >= tot + premium) and (sell_cpos_gb > -POSITION_LIMIT['GIFT_BASKET']):
                    order_for = min(POSITION_LIMIT['GIFT_BASKET'] + sell_cpos_gb,vol)
                    orders['GIFT_BASKET'].append(Order('GIFT_BASKET',bid,-order_for))
                    sell_cpos_gb += -order_for
        
        # market make for GIFT BASKET
        # if the price is little bit off I am willing to buy/sell just a bit more
        if (abs(res_buy) >= std*2):
            if (res_buy <= -std*2) and (buy_cpos_gb < POSITION_LIMIT['GIFT_BASKET']): #coupon underpriced, buy at competitive price
                order_for = POSITION_LIMIT['GIFT_BASKET'] - buy_cpos_gb
                bid_price = math.floor(mid_price['GIFT_BASKET'])+1
                orders['GIFT_BASKET'].append(Order(p,bid_price,order_for))
                buy_cpos_gb += order_for

            if (res_buy >= std*2) and (sell_cpos_gb > -POSITION_LIMIT['GIFT_BASKET']): #coupon overpriced, sell at competitive price
                order_for = POSITION_LIMIT['GIFT_BASKET'] + sell_cpos_gb
                ask_price = math.ceil(mid_price['GIFT_BASKET'])-1
                orders['GIFT_BASKET'].append(Order(p,ask_price,-order_for))
                sell_cpos_gb += -order_for

        # same strategy for chocolate
        std = 31.580057371497517
        res_cc = mid_price['CHOCOLATE'] -(-3098.200611 + 0.146460 * mid_price['GIFT_BASKET'] +0.045324 * mid_price['ROSES'])
        trade_at_cc = std * 0.2
        close_at_cc = std * (-0.2)
        if (res_cc > trade_at_cc): # overpriced, sell
            vol = state.position.get('CHOCOLATE',0) + mydata['position_limit']['CHOCOLATE']
            assert(vol >= 0)
            if vol > 0:
                orders['CHOCOLATE'].append(Order('CHOCOLATE', best_buy['CHOCOLATE'], -vol))
        elif (res_cc < -trade_at_cc):# underpriced, buy
            vol = mydata['position_limit']['CHOCOLATE'] - state.position.get('CHOCOLATE',0)
            assert(vol >= 0)
            if vol > 0:
                orders['CHOCOLATE'].append(Order('CHOCOLATE', best_sell['CHOCOLATE'], vol))

        elif res_cc < close_at_cc and state.position.get('CHOCOLATE',0) < 0: 
            vol = -state.position.get('CHOCOLATE',0)
            assert(vol >= 0)
            if vol > 0:
                orders['CHOCOLATE'].append(Order('CHOCOLATE', best_sell['CHOCOLATE'], vol))
        elif res_cc > -close_at_cc and state.position.get('CHOCOLATE',0) > 0:
            vol = state.position.get('CHOCOLATE',0)
            assert(vol >= 0)
            if vol > 0:
                orders['CHOCOLATE'].append(Order('CHOCOLATE', best_buy['CHOCOLATE'], -vol))

        # on round 5, when all bot traders names been disclosed
        # Rhianna makes consistent profit on roses, buy low sell high, copy Rhiannas trade
        for trade in (state.market_trades.get('ROSES',[])):
            logger.print(trade)
            cpos_rs = state.position.get('ROSES',0)
            if mydata['Rhianna_flag'] == 1:
                sell_cpos_rs = cpos_rs
                for bid, vol in obuy['ROSES']:
                    if bid == best_buy['ROSES']:
                        order_for = min(POSITION_LIMIT['ROSES'] + sell_cpos_rs,vol)
                        orders['ROSES'].append(Order('ROSES',bid,-order_for))
                        sell_cpos_rs += -order_for
            if mydata['Rhianna_flag'] == 0:
                buy_cpos_rs = cpos_rs
                for ask, vol in osell['ROSES']:
                    if ask == best_sell['ROSES']:
                        order_for = min(POSITION_LIMIT['ROSES'] - buy_cpos_rs,-vol)
                        #order_for = min(buy_cpos_rs,-vol)
                        orders['ROSES'].append(Order('ROSES',ask,order_for))
                        buy_cpos_rs += order_for
            
            if ((trade.buyer == 'Rhianna') or (trade.seller == 'Rhianna')) and (trade.timestamp == state.timestamp - 100):
                #copy rhianna
                rhianna_p = trade.price
                if trade.buyer == 'Rhianna':
                    # i also buy
                    mydata['Rhianna_flag'] = 0
                    buy_cpos_rs = cpos_rs
                    for ask, vol in osell['ROSES']:
                        #order_for = min(POSITION_LIMIT['ROSES'] - buy_cpos_rs,-vol)
                        order_for = min(buy_cpos_rs,-vol)
                        orders['ROSES'].append(Order('ROSES',ask,order_for))
                        buy_cpos_rs += order_for
                if trade.seller == 'Rhianna':
                    # i also sell
                    mydata['Rhianna_flag'] = 1
                    sell_cpos_rs = cpos_rs
                    for bid, vol in obuy['ROSES']:
                        order_for = min(POSITION_LIMIT['ROSES'] + sell_cpos_rs,vol)
                        orders['ROSES'].append(Order('ROSES',bid,-order_for))
                        sell_cpos_rs += -order_for

        return orders
        
    def compute_orders_ccnc(self, state: TradingState, mydata):

        orders = {'COCONUT' : [], 'COCONUT_COUPON': []}
        prods = ['COCONUT','COCONUT_COUPON']

        # for coconut price
        coef = 15.144511
        std = 395.47516494146237
        osell, obuy, best_sell, best_buy, worst_sell, worst_buy, mid_price, vol_buy, vol_sell = {}, {}, {}, {}, {}, {}, {}, {}, {}
        for p in prods:
            osell[p] = list(state.order_depths[p].sell_orders.items())
            obuy[p] = list(state.order_depths[p].buy_orders.items())
            best_buy[p] = 0
            if obuy[p]:
                best_buy[p], vol_buy[p] = obuy[p][0]
                worst_buy[p], dummy = obuy[p][-1]
            best_sell[p] = 0
            if osell[p]:
                best_sell[p], vol_sell[p] = osell[p][0]
                worst_sell[p], dummy = osell[p][-1]
            
            mid_price[p] = (best_sell[p] + best_buy[p])/2
            
        res = mid_price['COCONUT'] -coef*mid_price['COCONUT_COUPON'] - 17.18569706567527
        trade_at = std * 0.7
        close_at = std * (-0.7)

        if res > trade_at:
            vol = state.position.get('COCONUT',0) + mydata['position_limit']['COCONUT']
            assert(vol >= 0)
            if vol > 0:
                orders['COCONUT'].append(Order('COCONUT', worst_buy['COCONUT'], -vol))
        elif res < -trade_at:
            vol = mydata['position_limit']['COCONUT'] - state.position.get('COCONUT',0)
            assert(vol >= 0)
            if vol > 0:
                orders['COCONUT'].append(Order('COCONUT', worst_sell['COCONUT'], vol))
        elif res < close_at and state.position.get('COCONUT',0) < 0:
            vol = -state.position.get('COCONUT',0)
            assert(vol >= 0)
            if vol > 0:
                orders['COCONUT'].append(Order('COCONUT', worst_sell['COCONUT'], vol))
        elif res > -close_at and state.position.get('COCONUT',0) > 0:
            vol = state.position.get('COCONUT',0)
            assert(vol >= 0)
            if vol > 0:
                orders['COCONUT'].append(Order('COCONUT', worst_buy['COCONUT'], -vol))

        p = 'COCONUT'
        buy_orders_ccn = list(state.order_depths[p].buy_orders.items())
        best_buy_pr_ccn = 0
        if buy_orders_ccn:
            best_buy_pr_ccn, buy_vol_ccn = buy_orders_ccn[0]        
        sell_orders_ccn = list(state.order_depths[p].sell_orders.items())
        best_sell_pr_ccn = 0
        if sell_orders_ccn:
            best_sell_pr_ccn, sell_vol_ccn = sell_orders_ccn[0]
        if best_buy_pr_ccn == 0:
            ccn_mid_price = best_sell_pr_ccn
        elif best_sell_pr_ccn == 0:
            ccn_mid_price = best_buy_pr_ccn
        else:
            ccn_mid_price = (best_buy_pr_ccn + best_sell_pr_ccn)/2

        p = 'COCONUT_COUPON'
        
        buy_orders = list(state.order_depths[p].buy_orders.items())
        best_buy_pr = 0
        if buy_orders:
            best_buy_pr, buy_vol = buy_orders[0]
        sell_orders = list(state.order_depths[p].sell_orders.items())
        best_sell_pr = 0
        if sell_orders:
            best_sell_pr, sell_vol = sell_orders[0]
        if best_buy_pr == 0:
            mid_price = best_sell_pr
        elif best_sell_pr == 0:
            mid_price = best_buy_pr
        else:
            mid_price = (best_buy_pr + best_sell_pr)/2

        pos_limt = mydata['position_limit'][p]
        
        # use black scholes to estimate coconut coupon price 
        T = 250/365
        ccnc_price = self.black_scholes(S = ccn_mid_price, K = 10000, T = T, mydata = mydata)

        deviation = ccnc_price - mid_price - 1.88

        std = 0
        buy_cpos = state.position.get(p,0)
        buy_tot_order_for = 0
        if (deviation > std) and (buy_cpos < pos_limt): #coupon underpriced, buy now and sell later
            for ask, vol in sell_orders:
                if (ask <= ccnc_price - 1.88) and (buy_cpos < pos_limt):
                    order_for = min(pos_limt - buy_cpos,-vol)
                    buy_cpos += order_for
                    buy_tot_order_for += order_for
                    orders[p].append(Order(p,ask,order_for))
            
        sell_cpos = state.position.get(p,0)
        sell_tot_order_for = 0
        if (deviation < -std) and (sell_cpos > -pos_limt): #coupon overpriced, sell now and buy later
            for bid, vol in buy_orders:
                if (bid >= ccnc_price - 1.88) and (sell_cpos > -pos_limt):
                    order_for = min(pos_limt + sell_cpos,vol)
                    sell_cpos += -order_for
                    sell_tot_order_for += order_for
                    orders[p].append(Order(p,bid,-order_for))
                
        p = 'COCONUT_COUPON'
        std = 13.467
        # if the price is little bit off I am willing to buy/sell just a bit more
        if (abs(deviation) >= std) and (abs(deviation) < std*2):
            perc = 0.68
            if (deviation > std) and (buy_cpos < pos_limt): #coupon underpriced, buy at competitive price
                order_for = min(pos_limt - buy_cpos, int(perc*pos_limt))
                bid_price = min(best_buy_pr + 1, math.floor(mid_price))
                orders[p].append(Order(p,bid_price,order_for))
                buy_cpos += order_for

            if (deviation < -std) and (sell_cpos > -pos_limt): #coupon overpriced, sell at competitive price
                order_for = min(pos_limt + sell_cpos,int(perc*pos_limt))
                ask_price = max(best_sell_pr-1, math.ceil(mid_price))
                orders[p].append(Order(p,ask_price,-order_for))
                sell_cpos += -order_for
        # if the price is really off I am willing to buy/sell more
        if abs(deviation) >= std*2:
            perc = 0.87
            if (deviation > std) and (buy_cpos < pos_limt): #coupon underpriced, buy at competitive price
                order_for = min(pos_limt - buy_cpos, int(perc*pos_limt))
                bid_price = min(best_buy_pr + 1, math.floor(mid_price))
                orders[p].append(Order(p,bid_price,order_for))
                buy_cpos += order_for

            if (deviation < -std) and (sell_cpos > -pos_limt): #coupon overpriced, sell at competitive price
                order_for = min(pos_limt + sell_cpos,int(perc*pos_limt))
                ask_price = max(best_sell_pr-1, math.ceil(mid_price))
                orders[p].append(Order(p,ask_price,-order_for))
                sell_cpos += -order_for

            logger.print(f'ccnc_p;{ccnc_price};mid_p;{mid_price};deviation;{deviation};std;{std}')

        return orders

    def black_scholes(self, S, T, mydata, K = 10000, r = 0, sigma = 88.75*0.00217278, option="C"):
        """
        Calculates the Black-Scholes option price for calls or puts.

        Args:
            S (float): Spot price of the underlying asset.
            K (float): Strike price of the option.
            T (float): Time to expiration in years.
            r (float): Risk-free interest rate.
            sigma (float): Volatility of the underlying asset.
            option (str, optional): Option type ("C" for call, "P" for put). Defaults to "C".

        Returns:
            float: The Black-Scholes option price.
        """

        d1 = (np.log(S / K) + (r + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        d1 = min(max(d1, -3.0), 3.0)
        d2 = min(max(d2, -3.0), 3.0)

        if round(d1,2) == 0:
            d1 = abs(d1)
        if round(d2,2) == 0:
            d2 = abs(d2)
        if option == "C":
            price = S * mydata['cdf_table'][f'{round(d1,2)}'] - K * np.exp(-r * T) * mydata['cdf_table'][f'{round(d2,2)}']
        elif option == "P":
            price = K * np.exp(-r * T) * mydata['cdf_table'][f'{-round(d2,2)}'] - S * mydata['cdf_table'][f'{-round(d1,2)}']
        else:
            raise ValueError("Invalid option type. Please specify 'C' for call or 'P' for put.")

        return price
    